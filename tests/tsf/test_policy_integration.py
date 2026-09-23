"""TSF core/adapter contract tests without launching CoppeliaSim."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from essay2608.policy import DynaMAC, DynaMACObservation
from essay2608.policy.tsf import (
    ArmCommand,
    BoundaryRuntimeConfig,
    TSFMultiStreamPolicy,
    TSFFeatureProfile,
    ExecutionDecision,
    ExecutionMode,
    EntryGuard,
    PolicyLifecycle,
    ProgressStatus,
    RelationEventId,
    RelationVerificationRequest,
    RecoveryTriggerDecision,
    RuntimeObservation,
    StateId,
    TransitionPreparation,
)
from essay2608.policy.tsf.control.frame_roles import FrameRoleRouter
from essay2608.policy.tsf.inference.relation_filter import RelationDecision
from essay2608.policy.tsf.model.boundary_model import BoundaryId
from essay2608.policy.tsf.control.boundary_runtime import TransitionRequest
from integrations.rlbench.rlbench_tsf.observation_adapter import (
    TSFObservationAdapter,
    commands_to_rlbench,
)
from integrations.rlbench.rlbench_tsf.policy_server import (
    TSFPolicyServer,
)
from integrations.rlbench.iclr2027.task_registry import experiment_task
from integrations.rlbench.rlbench_dynamac.core.task_specs import get_task_spec
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    apply_gripper_for_policy_target,
    set_policy_gripper_authorization,
    policy_action_execution_status,
    policy_action_execution_statuses,
)
from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import demonstration_paths


ROOT = Path(__file__).resolve().parents[2]
UNIMANUAL_BASE_ROOT = ROOT / "integrations/rlbench/models/iclr2027/dynamac"
BIMANUAL_BASE_ROOT = ROOT / "integrations/rlbench/models/dynamac_backbone_v1"
BUNDLE_ROOT = ROOT / "integrations/rlbench/models/iclr2027/tsf"
UNIMANUAL_DATA_ROOT = ROOT / "integrations/rlbench/data/iclr2027/demonstrations"
BIMANUAL_DATA_ROOT = ROOT / "integrations/rlbench/data/training/main"
BOUNDARY_CONFIG_ROOT = (
    ROOT
    / "evaluations/iclr2027/artifacts/calibration/normal_task_boundaries/main10/runtime_configs"
)


def _boundary_config(task_id: str) -> BoundaryRuntimeConfig:
    return BoundaryRuntimeConfig.from_json(BOUNDARY_CONFIG_ROOT / f"{task_id}.json")


def test_tsf_ablation_profiles_add_mechanisms_monotonically() -> None:
    progress = TSFFeatureProfile.named("progress_only")
    roles = TSFFeatureProfile.named("progress_dynamic_roles")
    full = TSFFeatureProfile.named("full")

    assert progress.to_dict() == {
        "name": "progress_only",
        "dynamic_frame_roles": False,
        "relation_scene_boundary_guards": False,
        "active_relation_verification": False,
        "state_aware_recovery_reentry": False,
    }
    assert roles.dynamic_frame_roles is True
    assert roles.relation_scene_boundary_guards is False
    assert roles.active_relation_verification is False
    assert roles.state_aware_recovery_reentry is False
    assert full.dynamic_frame_roles is True
    assert full.relation_scene_boundary_guards is True
    assert full.active_relation_verification is True
    assert full.state_aware_recovery_reentry is True
    fine = {
        "no_relation_evidence": "relation_progress_evidence",
        "no_scene_evidence": "scene_progress_evidence",
        "static_stream_roles": "dynamic_frame_roles",
        "no_boundary_guards": "boundary_gated_advancement",
        "no_active_verification": "active_relation_verification",
        "retry_same_state": "legal_reentry_selection",
        "no_control_equivalence": "control_equivalence_aggregation",
    }
    assert set(fine).issubset(TSFFeatureProfile.names())
    for profile_name, disabled_field in fine.items():
        profile = TSFFeatureProfile.named(profile_name)
        assert getattr(profile, disabled_field) is False
        assert profile.state_aware_recovery_reentry is True
    assert (
        TSFFeatureProfile.named("retry_same_state").to_dict()[
            "legal_reentry_selection"
        ]
        is False
    )
    with pytest.raises(ValueError, match="unknown TSF feature profile"):
        TSFFeatureProfile.named("task_specific_shortcut")


def test_progress_only_bundle_load_disables_later_layers_without_changing_model() -> (
    None
):
    bundle = BUNDLE_ROOT / "stack_cups"
    checkpoint = UNIMANUAL_BASE_ROOT / "stack_cups/model.npz"
    if not bundle.is_dir() or not checkpoint.is_file():
        pytest.skip("本地未安装当前 StackCups v5 bundle 或 DynaMAC checkpoint")

    base = DynaMAC.load(checkpoint)
    policy = TSFMultiStreamPolicy.load(
        bundle,
        base_policies={"single": base},
        feature_profile="progress_only",
        boundary_config=_boundary_config("stack_cups"),
    )

    assert policy.feature_profile.name == "progress_only"
    assert policy.execution_controllers["single"].dynamic_frame_roles is False
    assert policy.boundary_controller.guards["single"].relation_scene_guards is False
    assert policy.summary()["feature_profile"] == policy.feature_profile.to_dict()


def test_fine_profile_wiring_changes_only_requested_controller_switches() -> None:
    bundle = BUNDLE_ROOT / "stack_cups"
    checkpoint = UNIMANUAL_BASE_ROOT / "stack_cups/model.npz"
    if not bundle.is_dir() or not checkpoint.is_file():
        pytest.skip("本地未安装当前 StackCups v5 bundle 或 DynaMAC checkpoint")

    base = DynaMAC.load(checkpoint)
    same_state = TSFMultiStreamPolicy.load(
        bundle,
        base_policies={"single": base},
        feature_profile="retry_same_state",
        boundary_config=_boundary_config("stack_cups"),
    )
    assert same_state.feature_profile.state_aware_recovery_reentry is True
    assert same_state.feature_profile.legal_reentry_selection is False
    assert same_state.execution_controllers["single"].dynamic_frame_roles is True

    no_equivalence = TSFMultiStreamPolicy.load(
        bundle,
        base_policies={"single": base},
        feature_profile="no_control_equivalence",
        boundary_config=_boundary_config("stack_cups"),
    )
    assert (
        no_equivalence.execution_controllers[
            "single"
        ].control_equivalence_aggregation
        is False
    )


def _pose_xyzw(x: float) -> np.ndarray:
    return np.asarray([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_direct_link_lifecycle_respects_mode_unlink_and_indirect_events() -> None:
    link = RelationEventId("left", "item", 0, 0, 0, "link")
    unlink = RelationEventId("left", "item", 1, 0, 0, "unlink")
    linked_state = StateId(0, 1)
    released_state = StateId(1, 1)
    router = object.__new__(FrameRoleRouter)
    router.task_model = SimpleNamespace(
        states={linked_state: object(), released_state: object()},
        link_anchors={
            link: SimpleNamespace(linked_entry_states=(linked_state,)),
        },
        unlink_events={
            unlink: SimpleNamespace(release_state=StateId(1, 0)),
        },
    )
    router._indirect_link_events = frozenset()

    assert router.active_direct_link_event(
        "item", linked_state, mode_by_skill={0: 0, 1: 0}
    ) == link
    assert router.selected_mode_unlink_event_in_skill(
        "item", 1, mode_by_skill={0: 0, 1: 0}
    ) == unlink
    assert router.selected_mode_unlink_event_in_skill(
        "item", 1, mode_by_skill={0: 0, 1: 1}
    ) is None
    assert router.active_direct_link_event(
        "item", linked_state, mode_by_skill={0: 1, 1: 0}
    ) is None
    assert router.active_direct_link_event(
        "item", released_state, mode_by_skill={0: 0, 1: 0}
    ) is None
    router._indirect_link_events = frozenset({link})
    assert router.active_direct_link_event(
        "item", linked_state, mode_by_skill={0: 0, 1: 0}
    ) is None


def test_auxiliary_shared_link_peer_preserves_relative_pose() -> None:
    policy = object.__new__(TSFMultiStreamPolicy)
    policy.arms = ("left", "right")
    policy._mode_by_arm_skill = {"left": {0: 0}, "right": {0: 0}}
    policy._shared_peer_relative = {}
    policy.recovery_managers = {
        "left": SimpleNamespace(mode=ExecutionMode.TASK),
        "right": SimpleNamespace(
            mode=ExecutionMode.VERIFY_LINK,
            verification=SimpleNamespace(
                request=SimpleNamespace(
                    frame_id="item",
                    event_id=RelationEventId(
                        "right", "item", 0, 0, 0, "link_pending"
                    ),
                )
            ),
        ),
    }
    event = RelationEventId("left", "item", 0, 0, 0, "link")
    role_router = SimpleNamespace(
        active_direct_link_event=lambda *args, **kwargs: event,
        selected_mode_unlink_event_in_skill=lambda *args, **kwargs: None,
    )
    policy.execution_controllers = {
        "left": SimpleNamespace(role_router=role_router),
        "right": SimpleNamespace(role_router=SimpleNamespace()),
    }
    beliefs = {
        "left": SimpleNamespace(
            relation_estimates={
                "item": SimpleNamespace(
                    decision_state=RelationDecision.LINKED
                )
            },
            progress=SimpleNamespace(estimated_state=StateId(0, 1)),
        ),
        "right": SimpleNamespace(),
    }
    covariance = np.eye(6)
    commands = {
        arm: ArmCommand(
            pose=_pose_xyzw(0.0),
            covariance=covariance,
            gripper=np.asarray([0.0]),
            source="auxiliary_peer_frozen_task_target",
        )
        for arm in policy.arms
    }
    runtime = {
        "left": SimpleNamespace(
            ee_pose=_pose_xyzw(0.2),
            frame_poses={"item": _pose_xyzw(0.1)},
        ),
        "right": SimpleNamespace(),
    }
    record = policy._coordinate_shared_linked_peers(commands, beliefs, runtime)
    assert record["applied"] == [
        {"recovering_arm": "right", "peer": "left", "frame": "item"}
    ]
    assert commands["left"].source == "auxiliary_shared_link_peer_follow"
    assert np.allclose(commands["left"].pose, _pose_xyzw(0.2))

    runtime["left"].frame_poses["item"] = _pose_xyzw(0.15)
    commands["left"] = replace(commands["left"], pose=_pose_xyzw(0.0))
    policy._coordinate_shared_linked_peers(commands, beliefs, runtime)
    assert np.allclose(commands["left"].pose, _pose_xyzw(0.25))

    policy.recovery_managers["right"].mode = ExecutionMode.TASK
    policy._coordinate_shared_linked_peers(commands, beliefs, runtime)
    assert policy._shared_peer_relative == {}


def test_auxiliary_shared_link_peer_does_not_oppose_ownership_transfer() -> None:
    policy = object.__new__(TSFMultiStreamPolicy)
    policy.arms = ("left", "right")
    policy._mode_by_arm_skill = {
        "left": {2: 0, 6: 0},
        "right": {6: 0},
    }
    policy._shared_peer_relative = {}
    recovering_event = RelationEventId(
        "right", "item", 6, 0, 0, "link_pending"
    )
    policy.recovery_managers = {
        "left": SimpleNamespace(mode=ExecutionMode.TASK),
        "right": SimpleNamespace(
            mode=ExecutionMode.VERIFY_LINK,
            verification=SimpleNamespace(
                request=SimpleNamespace(
                    frame_id="item", event_id=recovering_event
                )
            ),
        ),
    }
    left_link = RelationEventId("left", "item", 2, 0, 0, "link")
    left_unlink = RelationEventId("left", "item", 6, 0, 0, "unlink")
    role_router = SimpleNamespace(
        active_direct_link_event=lambda *args, **kwargs: left_link,
        selected_mode_unlink_event_in_skill=lambda *args, **kwargs: left_unlink,
    )
    policy.execution_controllers = {
        "left": SimpleNamespace(role_router=role_router),
        "right": SimpleNamespace(role_router=SimpleNamespace()),
    }
    beliefs = {
        "left": SimpleNamespace(
            relation_estimates={
                "item": SimpleNamespace(
                    decision_state=RelationDecision.LINKED
                )
            },
            progress=SimpleNamespace(estimated_state=StateId(2, 1)),
        ),
        "right": SimpleNamespace(),
    }
    commands = {
        arm: ArmCommand(
            pose=_pose_xyzw(0.0),
            covariance=np.eye(6),
            gripper=np.asarray([0.0]),
            source="auxiliary_peer_frozen_task_target",
        )
        for arm in policy.arms
    }
    runtime = {
        "left": SimpleNamespace(
            ee_pose=_pose_xyzw(0.2),
            frame_poses={"item": _pose_xyzw(0.1)},
        ),
        "right": SimpleNamespace(),
    }

    record = policy._coordinate_shared_linked_peers(commands, beliefs, runtime)

    assert record["applied"] == []
    assert record["ambiguous"] == []
    assert record["transfer_suppressed"] == [
        {
            "recovering_arm": "right",
            "peer": "left",
            "frame": "item",
            "recovering_link_event": recovering_event.token,
            "peer_link_event": left_link.token,
            "peer_unlink_event": left_unlink.token,
        }
    ]
    assert commands["left"].source == "auxiliary_peer_frozen_task_target"
    assert policy._shared_peer_relative == {}


def test_committed_pending_does_not_invent_second_owner() -> None:
    policy = object.__new__(TSFMultiStreamPolicy)
    policy.arms = ("left", "right")
    policy._mode_by_arm_skill = {
        "left": {1: 0, 2: 0},
        "right": {1: 0, 2: 0},
    }
    request = RelationVerificationRequest(
        arm_id="right",
        frame_id="bottle",
        relation="linked",
        event_id=RelationEventId(
            "right", "bottle", 1, 0, 0, "link_pending"
        ),
        context_state=StateId(1, 0),
        committed_occurrence=True,
    )
    left_link = RelationEventId("left", "bottle", 0, 0, 0, "link")
    policy.execution_controllers = {
        "left": SimpleNamespace(
            role_router=SimpleNamespace(
                active_direct_link_event=lambda *args, **kwargs: left_link,
                selected_mode_unlink_event_in_skill=lambda *args, **kwargs: None,
            )
        ),
        "right": SimpleNamespace(role_router=SimpleNamespace()),
    }
    beliefs = {
        "left": SimpleNamespace(
            relation_estimates={
                "bottle": SimpleNamespace(
                    decision_state=RelationDecision.LINKED
                )
            },
            progress=SimpleNamespace(estimated_state=StateId(1, 3)),
        ),
        "right": SimpleNamespace(),
    }

    assert policy._committed_pending_has_persistent_peer_owner(
        "right", request, beliefs
    )
    assert not policy._committed_pending_has_persistent_peer_owner(
        "right",
        replace(request, committed_occurrence=False),
        beliefs,
    )


def test_committed_pending_preserves_ownership_transfer_verification() -> None:
    policy = object.__new__(TSFMultiStreamPolicy)
    policy.arms = ("left", "right")
    policy._mode_by_arm_skill = {
        "left": {6: 0},
        "right": {6: 0},
    }
    request = RelationVerificationRequest(
        arm_id="right",
        frame_id="item",
        relation="linked",
        event_id=RelationEventId(
            "right", "item", 6, 0, 0, "link_pending"
        ),
        context_state=StateId(6, 0),
        committed_occurrence=True,
    )
    left_link = RelationEventId("left", "item", 2, 0, 0, "link")
    left_unlink = RelationEventId("left", "item", 6, 0, 0, "unlink")
    policy.execution_controllers = {
        "left": SimpleNamespace(
            role_router=SimpleNamespace(
                active_direct_link_event=lambda *args, **kwargs: left_link,
                selected_mode_unlink_event_in_skill=(
                    lambda *args, **kwargs: left_unlink
                ),
            )
        ),
        "right": SimpleNamespace(role_router=SimpleNamespace()),
    }
    beliefs = {
        "left": SimpleNamespace(
            relation_estimates={
                "item": SimpleNamespace(
                    decision_state=RelationDecision.LINKED
                )
            },
            progress=SimpleNamespace(estimated_state=StateId(2, 1)),
        ),
        "right": SimpleNamespace(),
    }

    assert not policy._committed_pending_has_persistent_peer_owner(
        "right", request, beliefs
    )


def test_rlbench_worker_reports_structured_policy_failure_before_next_tick() -> None:
    command = ArmCommand(
        pose=_pose_xyzw(0.1),
        covariance=np.eye(6),
        gripper=np.asarray([1.0]),
        source="structured_failure_hold",
    )
    cycle = SimpleNamespace(
        commands={"single": command},
        lifecycle=PolicyLifecycle.FAILED,
        arms={
            "single": SimpleNamespace(
                failure_reason="no_legal_reentry_state",
                mode_after=ExecutionMode.TASK,
            )
        },
    )
    batch = SimpleNamespace(
        dynamac={"single": object()},
        runtime={"single": SimpleNamespace(ee_pose=_pose_xyzw(0.0))},
    )
    server = object.__new__(TSFPolicyServer)
    server._pending = None
    server._next_transaction_id = 7
    server._previous_ee = {"single": None}
    server._previous_command = {"single": None}
    server._previous_command_covariance = {"single": None}
    server.arms = ("single",)
    server.bimanual = False
    server._adapt = lambda payload: batch
    server.policy = SimpleNamespace(act=lambda dynamac, runtime: cycle, complete=False)

    response = server._act({})

    assert response["policy_failed"] is True
    assert response["failure_reasons"] == {"single": "no_legal_reentry_state"}
    assert response["transaction_id"] == 7
    assert response["action"] is not None
    assert response["gripper_authorization"] == {"single": None}
    assert response["evaluator_audit"] == {
        "arms": {
            "single": {
                "mode_before": None,
                "mode_after": None,
                "reentry_committed": False,
            }
        }
    }


def test_rlbench_worker_exports_reentry_only_as_evaluator_audit() -> None:
    command = ArmCommand(
        pose=_pose_xyzw(0.1),
        covariance=np.eye(6),
        gripper=np.asarray([1.0]),
        source="reentry",
    )
    cycle = SimpleNamespace(
        commands={"single": command},
        lifecycle=PolicyLifecycle.RUNNING,
        arms={
            "single": SimpleNamespace(
                failure_reason=None,
                mode_after=ExecutionMode.TASK,
            )
        },
        diagnostics={
            "arms": {
                "single": {
                    "mode_before": "recovery",
                    "mode_after": "task",
                    "recovery": {"reentry": {"state_id": [2, 0]}},
                    "execution": {},
                }
            }
        },
    )
    batch = SimpleNamespace(
        dynamac={"single": object()},
        runtime={"single": SimpleNamespace(ee_pose=_pose_xyzw(0.0))},
    )
    server = object.__new__(TSFPolicyServer)
    server._pending = None
    server._next_transaction_id = 8
    server._previous_ee = {"single": None}
    server._previous_command = {"single": None}
    server._previous_command_covariance = {"single": None}
    server.arms = ("single",)
    server.bimanual = False
    server._adapt = lambda payload: batch
    server.policy = SimpleNamespace(
        act=lambda dynamac, runtime: cycle,
        complete=False,
        execution_controllers={},
    )

    response = server._act({})

    assert "evaluator_audit" not in response["policy_state"]
    assert response["evaluator_audit"] == {
        "arms": {
            "single": {
                "mode_before": "recovery",
                "mode_after": "task",
                "reentry_committed": True,
            }
        }
    }


def test_shadow_synchronizes_logged_boundary_as_action_context_only() -> None:
    source = StateId(0, 1)
    target = StateId(1, 0)
    boundary_id = BoundaryId("single", 0, 1)
    policy = object.__new__(TSFMultiStreamPolicy)
    policy.arms = ("single",)
    policy._initialized = True
    policy._pending_snapshot = {}
    policy._permitted_boundaries = {"single": frozenset()}
    policy.task_models = {
        "single": SimpleNamespace(
            states={source: object(), target: object()},
            boundaries={
                boundary_id: SimpleNamespace(transaction_group=None),
            },
        )
    }
    policy.synchronize_observer_transition_context(
        {"single": source},
        {"single": target},
    )

    # This is action-prior context only.  It neither commits a transaction nor
    # creates an additional monitor alarm path.
    assert policy._permitted_boundaries == {"single": frozenset({boundary_id})}


def test_rlbench_worker_commits_bounded_progress_without_completing_target() -> None:
    class Diagnostics:
        def __init__(self) -> None:
            self.annotation = None

        def annotate_last(self, name, value) -> None:
            self.annotation = (name, value)

    class Policy:
        def __init__(self) -> None:
            self.complete = False
            self.commits = []
            self.diagnostics = Diagnostics()

        def commit(
            self,
            *,
            task_command_applied,
            absolute_target_completed,
        ) -> None:
            self.commits.append((task_command_applied, absolute_target_completed))

    server = object.__new__(TSFPolicyServer)
    server.arms = ("single",)
    server.policy = Policy()
    server._tick = 4
    server._previous_ee = {"single": None}
    server._previous_command = {"single": None}
    server._previous_command_covariance = {"single": None}
    pre_action = _pose_xyzw(0.0)
    target = _pose_xyzw(0.2)
    covariance = np.eye(6)
    server._pending = {
        "transaction_id": 9,
        "pre_action_ee": {"single": pre_action},
        "commands": {"single": target},
        "command_covariances": {"single": covariance},
        "gripper_authorization": {"single": None},
    }

    response = server._resolve(
        {"transaction_id": 9, "primary_action_status": "progressed"},
        commit=True,
    )

    assert server.policy.commits == [({"single": True}, {"single": False})]
    assert server._tick == 5
    assert np.array_equal(server._previous_ee["single"], pre_action)
    assert np.array_equal(server._previous_command["single"], target)
    assert np.array_equal(server._previous_command_covariance["single"], covariance)
    assert response["primary_action_status"] == "progressed"
    assert response["complete"] is False
    assert server.policy.diagnostics.annotation == (
        "rlbench_action_resolution",
        {
            "status": "progressed",
            "status_by_arm": {"single": "progressed"},
            "primary_action_applied": True,
            "task_command_applied": {"single": True},
            "action_response_observed": {"single": True},
            "absolute_target_completed": {"single": False},
            "gripper_authorization": {"single": None},
        },
    )


def test_rlbench_worker_does_not_use_bimanual_status_as_progress_commit() -> None:
    class Diagnostics:
        def annotate_last(self, name, value) -> None:
            del name, value

    class Policy:
        def __init__(self) -> None:
            self.complete = False
            self.commit_args = None
            self.diagnostics = Diagnostics()

        def commit(
            self,
            *,
            task_command_applied,
            absolute_target_completed,
        ) -> None:
            self.commit_args = (
                task_command_applied,
                absolute_target_completed,
            )

    server = object.__new__(TSFPolicyServer)
    server.arms = ("left", "right")
    server.policy = Policy()
    server._tick = 2
    pose = _pose_xyzw(0.0)
    target = _pose_xyzw(0.1)
    server._previous_ee = {"left": None, "right": None}
    server._previous_command = {"left": None, "right": None}
    server._previous_command_covariance = {"left": None, "right": None}
    server._pending = {
        "transaction_id": 3,
        "pre_action_ee": {"left": pose.copy(), "right": pose.copy()},
        "commands": {"left": target.copy(), "right": target.copy()},
        "command_covariances": {"left": np.eye(6), "right": np.eye(6)},
        "gripper_authorization": {"left": None, "right": None},
    }

    response = server._resolve(
        {
            "transaction_id": 3,
            "primary_action_status": "progressed",
            "primary_action_statuses": {
                "left": "stopped",
                "right": "progressed",
            },
        },
        commit=True,
    )

    assert server.policy.commit_args == (
        {"left": True, "right": True},
        {"left": False, "right": False},
    )
    assert response["primary_action_statuses"] == {
        "left": "stopped",
        "right": "progressed",
    }


def test_rlbench_worker_uses_explicit_stopped_unapplied_primary_action() -> None:
    class Diagnostics:
        def __init__(self) -> None:
            self.annotation = None

        def annotate_last(self, name, value) -> None:
            self.annotation = (name, value)

    class Policy:
        def __init__(self) -> None:
            self.complete = False
            self.commit_args = None
            self.diagnostics = Diagnostics()

        def commit(self, *, task_command_applied, absolute_target_completed) -> None:
            self.commit_args = (task_command_applied, absolute_target_completed)

    server = object.__new__(TSFPolicyServer)
    server.arms = ("single",)
    server.policy = Policy()
    server._tick = 2
    pose = _pose_xyzw(0.0)
    target = _pose_xyzw(0.1)
    server._previous_ee = {"single": None}
    server._previous_command = {"single": None}
    server._previous_command_covariance = {"single": None}
    server._pending = {
        "transaction_id": 4,
        "pre_action_ee": {"single": pose},
        "commands": {"single": target},
        "command_covariances": {"single": np.eye(6)},
        "gripper_authorization": {"single": None},
    }

    response = server._resolve(
        {
            "transaction_id": 4,
            "primary_action_status": "stopped",
            "primary_action_applied": False,
        },
        commit=True,
    )

    assert server.policy.commit_args == ({"single": False}, {"single": False})
    assert np.array_equal(server._previous_command["single"], pose)
    assert server._previous_command_covariance["single"] is None
    assert response["primary_action_applied"] is False
    assert server.policy.diagnostics.annotation[1]["primary_action_applied"] is False


def test_rlbench_worker_rejects_unknown_commit_fields() -> None:
    server = object.__new__(TSFPolicyServer)
    server.arms = ("single",)
    server._pending = {"transaction_id": 5}

    with pytest.raises(ValueError, match="闭环动作事务包含未知字段"):
        server._resolve(
            {"transaction_id": 5, "retired_flag": False},
            commit=True,
        )


def test_rlbench_action_status_reads_hybrid_arm_mode_only_when_available() -> None:
    assert policy_action_execution_status(SimpleNamespace()) == "reached"
    assert policy_action_execution_statuses(SimpleNamespace()) == {"single": "reached"}
    environment = SimpleNamespace(
        _action_mode=SimpleNamespace(
            arm_action_mode=SimpleNamespace(
                policy_action_status=lambda: "progressed",
                policy_action_statuses=lambda: {
                    "left": "stopped",
                    "right": "progressed",
                },
            )
        )
    )
    assert policy_action_execution_status(environment) == "progressed"
    assert policy_action_execution_statuses(environment) == {
        "left": "stopped",
        "right": "progressed",
    }
    environment._action_mode.arm_action_mode.policy_action_status = lambda: "invalid"
    with pytest.raises(RuntimeError, match="unsupported policy action"):
        policy_action_execution_status(environment)


def test_hybrid_gripper_uses_task_permission_without_a_second_pose_gate() -> None:
    calls = []
    gripper = SimpleNamespace(
        action=lambda scene, action: calls.append((scene, action.copy()))
    )
    scene = object()
    command = np.asarray([1.0])

    for status in ("progressed", "stopped"):
        assert not apply_gripper_for_policy_target(
            gripper,
            scene,
            command,
            arm_status=status,
        )
    assert calls == []

    assert apply_gripper_for_policy_target(
        gripper,
        scene,
        command,
        arm_status="reached",
    )
    assert len(calls) == 1
    assert calls[0][0] is scene
    assert np.array_equal(calls[0][1], command)

    assert apply_gripper_for_policy_target(
        gripper,
        scene,
        command,
        arm_status="stopped",
        gripper_authorized=True,
    )
    assert apply_gripper_for_policy_target(
        gripper,
        scene,
        command,
        arm_status="reached",
        gripper_authorized=True,
    )
    assert not apply_gripper_for_policy_target(
        gripper,
        scene,
        command,
        arm_status="reached",
        gripper_authorized=False,
    )
    assert len(calls) == 3


def test_task_gripper_authorization_uses_alignment_and_boundary_transaction() -> None:
    source = StateId(0, 2)
    entry = StateId(1, 0)
    final = StateId(1, 3)
    policy = SimpleNamespace(
        task_models={
            "right": SimpleNamespace(
                states={
                    source: SimpleNamespace(
                        topology=SimpleNamespace(has_cross_skill_successor=True)
                    ),
                    entry: SimpleNamespace(
                        topology=SimpleNamespace(has_cross_skill_successor=False)
                    ),
                    final: SimpleNamespace(
                        topology=SimpleNamespace(has_cross_skill_successor=False)
                    ),
                }
            )
        },
        _committed_gripper_commands={"right": None},
    )
    command = ArmCommand(
        pose=_pose_xyzw(0.1),
        covariance=np.eye(6),
        gripper=np.asarray([-1.0]),
        source="task_poe",
    )

    def authorization(reference, estimated, status, boundary=None):
        return TSFMultiStreamPolicy._task_gripper_authorized(
            policy,
            "right",
            command,
            SimpleNamespace(cursor_after=SimpleNamespace(reference_state=reference)),
            SimpleNamespace(
                progress=SimpleNamespace(
                    status=status,
                    estimated_state=estimated,
                )
            ),
            boundary,
            SimpleNamespace(gripper_state=np.asarray([0.0])),
        )

    # Ordinary entry-state gripper commands remain transaction-gated.  A
    # learned relation-establishing close is handled separately by the
    # boundary transition preparation tested below; it never changes this
    # authorization.
    assert authorization(source, source, ProgressStatus.ALIGNED) is False
    # A committed boundary authorizes the entry-state command in the same
    # cycle although the posterior still describes the source terminal.
    committed = SimpleNamespace(
        transaction=SimpleNamespace(committed=(SimpleNamespace(arm_id="right"),))
    )
    assert authorization(entry, source, ProgressStatus.ALIGNED, committed) is True
    # An ordinary/final task state is authorized only by posterior/reference
    # agreement; physical reached/progressed/stopped is not an input.
    assert authorization(final, final, ProgressStatus.ALIGNED) is True
    assert authorization(final, entry, ProgressStatus.ALIGNED) is False
    assert authorization(final, final, ProgressStatus.LOW_CONFIDENCE) is False


def test_same_skill_advance_atomically_authorizes_discrete_gripper_change() -> None:
    source = StateId(0, 2)
    target = StateId(0, 3)
    nodes = {
        source: SimpleNamespace(
            topology=SimpleNamespace(
                has_cross_skill_successor=False,
                successors=(target,),
            )
        ),
        target: SimpleNamespace(
            topology=SimpleNamespace(
                has_cross_skill_successor=False,
                successors=(),
            )
        ),
    }
    model = SimpleNamespace(states=nodes, state=lambda state: nodes[state])
    policy = SimpleNamespace(
        task_models={"right": model},
        _committed_gripper_commands={"right": np.asarray([-1.0])},
        _last_observed_gripper_states={"right": np.asarray([0.0])},
    )
    execution = SimpleNamespace(
        decision=ExecutionDecision.ADVANCE,
        cursor_before=SimpleNamespace(reference_state=source),
        cursor_after=SimpleNamespace(reference_state=target),
    )
    command = ArmCommand(
        pose=_pose_xyzw(0.1),
        covariance=np.eye(6),
        gripper=np.asarray([1.0]),
        source="task_poe",
    )
    belief = SimpleNamespace(
        progress=SimpleNamespace(
            status=ProgressStatus.BACKWARD_REALIGNMENT,
            estimated_state=source,
        )
    )

    assert TSFMultiStreamPolicy._task_gripper_authorized(
        policy,
        "right",
        command,
        execution,
        belief,
        None,
        SimpleNamespace(gripper_state=np.asarray([0.0])),
    ) is True


def test_newly_lost_committed_gripper_state_is_reasserted_only_on_the_edge(
) -> None:
    current = StateId(0, 2)
    policy = SimpleNamespace(
        task_models={
            "right": SimpleNamespace(
                states={
                    current: SimpleNamespace(
                        topology=SimpleNamespace(has_cross_skill_successor=False)
                    )
                }
            )
        },
        _committed_gripper_commands={"right": np.asarray([-1.0])},
        _last_observed_gripper_states={"right": np.asarray([-1.0])},
    )
    execution = SimpleNamespace(
        cursor_after=SimpleNamespace(reference_state=current)
    )
    lagging_belief = SimpleNamespace(
        progress=SimpleNamespace(
            status=ProgressStatus.ALIGNED,
            estimated_state=StateId(0, 1),
        )
    )
    closed = ArmCommand(
        pose=_pose_xyzw(0.1),
        covariance=np.eye(6),
        gripper=np.asarray([-1.0]),
        source="task_poe",
    )
    opened = replace(closed, gripper=np.asarray([1.0]))
    physically_open = SimpleNamespace(gripper_state=np.asarray([1.0]))
    physically_closed = SimpleNamespace(gripper_state=np.asarray([0.0]))

    # A newly lost committed close is actuator-state maintenance rather than a
    # new task transition, so the loss edge does not acquire a pose gate.
    assert TSFMultiStreamPolicy._task_gripper_authorized(
        policy,
        "right",
        closed,
        execution,
        lagging_belief,
        None,
        physically_open,
    ) is True
    # The same persistent mismatch must not re-run a potentially blocking
    # gripper action on every control cycle.  Contact can legitimately keep
    # aperture on the opposite side of the binary observation threshold.
    policy._last_observed_gripper_states["right"] = np.asarray([1.0])
    assert not TSFMultiStreamPolicy._task_gripper_authorized(
        policy,
        "right",
        closed,
        execution,
        lagging_belief,
        None,
        physically_open,
    )
    # A mismatch that was never preceded by an observed committed state is
    # likewise not evidence of an external reversal.
    policy._last_observed_gripper_states["right"] = None
    assert not TSFMultiStreamPolicy._task_gripper_authorized(
        policy,
        "right",
        closed,
        execution,
        lagging_belief,
        None,
        physically_open,
    )
    # A satisfied actuator state follows the unmodified posterior path and is
    # not repeatedly actuated during nominal execution.
    assert not TSFMultiStreamPolicy._task_gripper_authorized(
        policy,
        "right",
        closed,
        execution,
        lagging_belief,
        None,
        physically_closed,
    )
    # A different binary command remains subject to the ordinary progress
    # authorization and therefore cannot be applied early by this probe.
    assert not TSFMultiStreamPolicy._task_gripper_authorized(
        policy,
        "right",
        opened,
        execution,
        lagging_belief,
        None,
        physically_closed,
    )


def test_learned_boundary_link_close_is_a_candidate_without_committing_progress() -> None:
    source = StateId(5, 21)
    earlier = StateId(5, 18)
    entry = StateId(6, 0)
    boundary_id = BoundaryId("right", 5, 6)
    event_id = RelationEventId(
        arm_id="right",
        frame_id="item",
        skill_index=6,
        mode=0,
        occurrence=0,
        transition="link_pending",
    )
    boundary_model = SimpleNamespace(
        boundary_id=boundary_id,
        terminal_window=(StateId(5, 19), source),
    )
    nodes = {
        source: SimpleNamespace(gripper_commands=np.asarray([[1.0]])),
        entry: SimpleNamespace(gripper_commands=np.asarray([[-1.0]])),
    }
    model = SimpleNamespace(
        link_pending_events={
            event_id: SimpleNamespace(candidate_state=entry),
        },
        link_anchors={},
        state=lambda state: nodes[state],
    )
    guard = SimpleNamespace(task_model=model, relation_scene_guards=True)
    request = TransitionRequest(
        tick=7,
        arm_id="right",
        boundary_id=boundary_id,
        permitted=False,
        source_state=source,
        target_state=entry,
        local_done=False,
        condition_results={},
    )
    belief = SimpleNamespace(
        progress=SimpleNamespace(estimated_state=source),
    )

    preparation = EntryGuard._transition_preparation(
        guard,
        boundary=boundary_model,
        source_state=source,
        target_state=entry,
        belief=belief,
        source_mode=0,
        target_mode=0,
        guard_results=[],
    )
    assert preparation is not None
    assert preparation.boundary_id == boundary_id
    assert preparation.event_ids == (event_id,)
    assert np.array_equal(preparation.gripper_command, np.asarray([-1.0]))
    assert request.source_state == source

    # Candidate generation alone does not authorize execution.  The shared
    # boundary controller additionally requires an active entry guard that
    # explicitly depends on this exact LINK relation.

    # The two progress ablations do not receive this later-stage
    # relation/boundary mechanism.
    guard.relation_scene_guards = False
    assert (
        EntryGuard._transition_preparation(
            guard,
            boundary=boundary_model,
            source_state=source,
            target_state=entry,
            belief=belief,
            source_mode=0,
            target_mode=0,
            guard_results=[],
        )
        is None
    )
    guard.relation_scene_guards = True

    # The same close is not prepared before the terminal window, and removing
    # its learned LINK/PENDING support removes the exception entirely.
    early_belief = SimpleNamespace(
        progress=SimpleNamespace(estimated_state=earlier),
    )
    assert (
        EntryGuard._transition_preparation(
            guard,
            boundary=boundary_model,
            source_state=source,
            target_state=entry,
            belief=early_belief,
            source_mode=0,
            target_mode=0,
            guard_results=[],
        )
        is None
    )
    model.link_pending_events = {}
    assert (
        EntryGuard._transition_preparation(
            guard,
            boundary=boundary_model,
            source_state=source,
            target_state=entry,
            belief=belief,
            source_mode=0,
            target_mode=0,
            guard_results=[],
        )
        is None
    )


def test_boundary_transition_preparation_never_preexecutes_release() -> None:
    source = StateId(2, 9)
    entry = StateId(3, 0)
    boundary_id = BoundaryId("left", 2, 3)
    event_id = RelationEventId(
        arm_id="left",
        frame_id="item",
        skill_index=3,
        mode=0,
        occurrence=0,
        transition="link_pending",
    )
    nodes = {
        source: SimpleNamespace(gripper_commands=np.asarray([[-1.0]])),
        entry: SimpleNamespace(gripper_commands=np.asarray([[1.0]])),
    }
    boundary_model = SimpleNamespace(
        boundary_id=boundary_id,
        terminal_window=(source,),
    )
    model = SimpleNamespace(
        link_pending_events={event_id: SimpleNamespace(candidate_state=entry)},
        link_anchors={},
        state=lambda state: nodes[state],
    )
    guard = SimpleNamespace(task_model=model, relation_scene_guards=True)
    belief = SimpleNamespace(progress=SimpleNamespace(estimated_state=source))
    assert (
        EntryGuard._transition_preparation(
            guard,
            boundary=boundary_model,
            source_state=source,
            target_state=entry,
            belief=belief,
            source_mode=0,
            target_mode=0,
            guard_results=[],
        )
        is None
    )


def test_current_state_gripper_transition_must_commit_before_successor() -> None:
    current = StateId(0, 2)
    terminal = StateId(1, 3)
    policy = SimpleNamespace(
        execution_controllers={
            "right": SimpleNamespace(cursor=SimpleNamespace(reference_state=current))
        },
        task_models={
            "right": SimpleNamespace(
                state=lambda state: {
                    current: SimpleNamespace(
                        topology=SimpleNamespace(has_cross_skill_successor=False),
                        gripper_commands=np.asarray([[-1.0]]),
                    ),
                    terminal: SimpleNamespace(
                        topology=SimpleNamespace(has_cross_skill_successor=True),
                        gripper_commands=np.asarray([[-1.0]]),
                    ),
                }[state],
                base_policy=SimpleNamespace(selected_mode_path=(0, 0)),
            )
        },
        _mode_by_arm_skill={"right": {0: 0, 1: 0}},
        _committed_gripper_commands={"right": None},
    )
    open_observation = SimpleNamespace(gripper_state=np.asarray([1.0]))
    closed_observation = SimpleNamespace(gripper_state=np.asarray([0.0]))

    assert not TSFMultiStreamPolicy._current_discrete_action_committed(
        policy, "right", open_observation
    )
    assert TSFMultiStreamPolicy._current_discrete_action_committed(
        policy, "right", closed_observation
    )
    # Once the close command has been accepted by the execution transaction,
    # contact may keep the observed aperture near its open threshold without
    # turning command sequencing into a second physical-outcome gate.
    policy._committed_gripper_commands["right"] = np.asarray([-1.0])
    assert TSFMultiStreamPolicy._current_discrete_action_committed(
        policy, "right", open_observation
    )

    # A non-final skill terminal is owned by the boundary transaction and must
    # not deadlock while waiting for its entry-state gripper command.
    policy.execution_controllers["right"].cursor.reference_state = terminal
    assert TSFMultiStreamPolicy._current_discrete_action_committed(
        policy, "right", open_observation
    )

    # Once a learned edge preparation has closed the gripper, the source
    # node's pre-transition open label must not reopen it or block continuous
    # progress through the remainder of the terminal window.
    preparation_source = StateId(0, 1)
    preparation_boundary = BoundaryId("right", 0, 1)
    preparation = TransitionPreparation(
        boundary_id=preparation_boundary,
        event_ids=(
            RelationEventId(
                arm_id="right",
                frame_id="item",
                skill_index=1,
                mode=0,
                occurrence=0,
                transition="link_pending",
            ),
        ),
        gripper_command=np.asarray([-1.0]),
    )
    prepared_policy = SimpleNamespace(
        execution_controllers={
            "right": SimpleNamespace(
                cursor=SimpleNamespace(reference_state=preparation_source)
            )
        },
        boundary_controller=SimpleNamespace(
            preparations={"right": preparation}
        ),
        task_models={
            "right": SimpleNamespace(
                boundaries={
                    preparation_boundary: SimpleNamespace(
                        terminal_window=(preparation_source,)
                    )
                },
                state=lambda _state: SimpleNamespace(
                    topology=SimpleNamespace(has_cross_skill_successor=False),
                    gripper_commands=np.asarray([[1.0]]),
                ),
            )
        },
        _mode_by_arm_skill={"right": {0: 0}},
        _committed_gripper_commands={"right": None},
    )
    assert TSFMultiStreamPolicy._current_discrete_action_committed(
        prepared_policy, "right", closed_observation
    )
    assert not TSFMultiStreamPolicy._current_discrete_action_committed(
        prepared_policy, "right", open_observation
    )


def test_rlbench_gripper_authorization_is_forwarded_to_action_mode() -> None:
    received = []
    environment = SimpleNamespace(
        _action_mode=SimpleNamespace(
            set_policy_gripper_authorization=lambda value: received.append(value)
        )
    )
    authorization = {"left": False, "right": True}
    set_policy_gripper_authorization(environment, authorization)
    assert received == [authorization]

    with pytest.raises(RuntimeError, match="cannot consume"):
        set_policy_gripper_authorization(
            SimpleNamespace(),
            {"single": True},
        )


def test_directional_boundary_verification_request_routes_to_receiver_arm() -> None:
    event_id = RelationEventId("right", "item0", 6, 0, 0, "link_pending")
    request = RelationVerificationRequest(
        arm_id="right",
        frame_id="item0",
        relation="linked",
        event_id=event_id,
        context_state=StateId(6, 0),
    )
    boundary = SimpleNamespace(
        requests={
            "left": SimpleNamespace(verification_requests=(request,)),
            "right": SimpleNamespace(verification_requests=()),
        }
    )
    policy = SimpleNamespace(
        _committed_gripper_commands={"left": None, "right": None}
    )
    selected = TSFMultiStreamPolicy._verification_request(
        policy, "right", None, boundary
    )
    assert selected == request
    assert (
        TSFMultiStreamPolicy._verification_request(
            policy, "left", None, boundary
        )
        is None
    )


def test_boundary_preparation_pending_request_waits_for_committed_close() -> None:
    event_id = RelationEventId("right", "item0", 6, 0, 0, "link_pending")
    request = RelationVerificationRequest(
        arm_id="right",
        frame_id="item0",
        relation="linked",
        event_id=event_id,
        context_state=StateId(6, 0),
    )
    preparation = TransitionPreparation(
        boundary_id=BoundaryId("right", 5, 6),
        event_ids=(event_id,),
        gripper_command=np.asarray([-1.0]),
    )
    boundary = SimpleNamespace(
        requests={
            "right": SimpleNamespace(verification_requests=(request,)),
        },
        preparations={"right": preparation},
    )
    policy = SimpleNamespace(_committed_gripper_commands={"right": None})
    assert TSFMultiStreamPolicy._verification_request(
        policy, "right", None, boundary
    ) is None
    policy._committed_gripper_commands["right"] = np.asarray([-1.0])
    assert (
        TSFMultiStreamPolicy._verification_request(
            policy,
            "right",
            None,
            boundary,
        )
        == request
    )


def test_rlbench_adapter_preserves_shared_snapshot_and_opposite_ee_frames() -> None:
    spec = get_task_spec("bimanual_handover_item")
    task_state = np.concatenate(
        [_pose_xyzw(float(index)) for index in range(len(spec.pose_chunks))]
    )
    payload = {
        "left": {"gripper_pose": _pose_xyzw(10.0), "gripper_open": 1.0},
        "right": {"gripper_pose": _pose_xyzw(20.0), "gripper_open": 0.0},
        "task_low_dim_state": task_state,
    }
    adapter = TSFObservationAdapter(spec)
    batch = adapter.build(
        payload,
        tick=3,
        previous_ee_pose={"left": None, "right": None},
        previous_command_pose={"left": None, "right": None},
    )
    assert np.allclose(batch.dynamac["left"].frames["right_ee"][:3], [20, 0, 0])
    assert np.allclose(batch.dynamac["right"].frames["left_ee"][:3], [10, 0, 0])
    assert np.allclose(batch.runtime["left"].frame_poses["left_ee"][:3], [10, 0, 0])
    assert np.allclose(batch.runtime["right"].frame_poses["right_ee"][:3], [20, 0, 0])
    assert batch.runtime["left"].gripper_state.tolist() == [1.0]
    assert batch.runtime["right"].gripper_state.tolist() == [-1.0]
    assert all(batch.runtime["left"].frame_visibility.values())
    assert all(
        value == 1.0 for value in batch.runtime["right"].tracking_reliability.values()
    )

    command = ArmCommand(
        pose=np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(6),
        gripper=np.asarray([1.0]),
        source="test",
    )
    wire = commands_to_rlbench(
        {"left": command, "right": command},
        bimanual=True,
    )
    assert wire.shape == (18,)
    assert wire[7] == wire[16] == 1.0


def test_rlbench_adapter_preserves_the_upstream_wipe_task_frame() -> None:
    spec = get_task_spec("wipe_desk")
    task_state = _pose_xyzw(1.0)
    batch = TSFObservationAdapter(spec).build(
        {
            "gripper_pose": _pose_xyzw(0.0),
            "gripper_open": 1.0,
            "task_low_dim_state": task_state,
        },
        tick=1,
        previous_ee_pose={"single": None},
        previous_command_pose={"single": None},
    )
    assert tuple(batch.dynamac["single"].frames) == spec.action_frame_names
    assert tuple(batch.dynamac["single"].frames) == ("sponge",)
    assert set(batch.runtime["single"].frame_poses) == {
        "sponge",
        "single_ee",
    }
    assert batch.runtime["single"].entity_configurations == {}


def test_top_policy_bootstrap_abort_and_physical_posterior_controls_progress() -> None:
    bundle = BUNDLE_ROOT / "stack_cups"
    checkpoint = UNIMANUAL_BASE_ROOT / "stack_cups/model.npz"
    try:
        path = demonstration_paths(UNIMANUAL_DATA_ROOT, "stack_cups", 1)[0]
    except FileNotFoundError:
        pytest.skip("本地未安装当前 StackCups 正常示范")
    if not bundle.is_dir() or not checkpoint.is_file():
        pytest.skip("本地未安装当前 StackCups v6 bundle 或 DynaMAC checkpoint")

    base = DynaMAC.load(checkpoint)
    policy = TSFMultiStreamPolicy.load(
        bundle,
        base_policies={"single": base},
        boundary_config=_boundary_config("stack_cups"),
    )
    episode = load_low_dim_obs_pickles([path])[0]
    converted = make_unimanual_demonstrations(
        [episode],
        experiment_task("stack_cups").spec,
        names=[path.parent.name],
    ).demonstrations[0]
    dynamac = DynaMACObservation(
        converted.ee_pose[0],
        {name: values[0] for name, values in converted.frames.items()},
    )
    runtime0 = RuntimeObservation.from_dynamac(
        dynamac,
        tick=0,
        gripper_state=converted.gripper[0],
    )
    policy.reset({"single": dynamac})
    first = policy.act({"single": dynamac}, {"single": runtime0})
    initial = min(policy.task_models["single"].states)
    assert first.arms["single"].execution is not None
    assert first.arms["single"].execution.decision == ExecutionDecision.HOLD
    assert first.arms["single"].belief.progress.nominal_state == initial
    assert first.arms["single"].belief.progress.estimated_state == initial
    policy.abort()
    assert not policy.pending
    assert policy.diagnostics.records == ()

    rejected = policy.act({"single": dynamac}, {"single": runtime0})
    assert rejected.arms["single"].belief.progress.nominal_state == initial
    policy.commit(task_command_applied=False)
    assert policy.diagnostics.records[-1]["action_commit"] == {
        "task_command_applied": {"single": False},
        "absolute_target_completed": {"single": False},
        "gripper_command_applied": {"single": False},
        "executed_reference_states": None,
    }
    arm_diagnostics = policy.diagnostics.records[-1]["arms"]["single"]
    assert "actual_ee_motion" in arm_diagnostics["motion_features"]
    relation_diagnostics = next(iter(arm_diagnostics["relations"].values()))
    assert "predicted" in relation_diagnostics
    assert "observation_likelihood" in relation_diagnostics
    runtime1 = RuntimeObservation.from_dynamac(
        dynamac,
        tick=1,
        gripper_state=converted.gripper[0],
        previous_ee_pose=dynamac.ee_pose,
        previous_command_pose=dynamac.ee_pose,
    )
    after_hold = policy.act({"single": dynamac}, {"single": runtime1})
    assert after_hold.arms["single"].belief.progress.nominal_state == initial
    assert after_hold.arms["single"].execution is not None
    # Executor acceptance is physical feedback, not a second task-progress
    # gate.  Once a later observation still explains the current target, the
    # posterior may advance the action reference even if the previous command
    # was rejected.  The bootstrap cycle above remains a HOLD because it has
    # no preceding observation interval.
    successor = policy.task_models["single"].state(initial).topology.successors[0]
    assert after_hold.arms["single"].execution.cursor_after.reference_state == successor
    policy.abort()
    replay = policy.act({"single": dynamac}, {"single": runtime1})
    assert replay.arms["single"].belief.progress.nominal_state == initial
    policy.commit(executed_reference_states={"single": initial})
    assert policy._last_executed_reference == {"single": initial}


def test_policy_completes_after_authorized_final_gripper_command_is_committed() -> None:
    bundle = BUNDLE_ROOT / "stack_cups"
    checkpoint = UNIMANUAL_BASE_ROOT / "stack_cups/model.npz"
    try:
        path = demonstration_paths(UNIMANUAL_DATA_ROOT, "stack_cups", 1)[0]
    except FileNotFoundError:
        pytest.skip("本地未安装当前 StackCups 正常示范")
    if not bundle.is_dir() or not checkpoint.is_file():
        pytest.skip("本地未安装当前 StackCups v5 bundle 或 DynaMAC checkpoint")

    base = DynaMAC.load(checkpoint)
    policy = TSFMultiStreamPolicy.load(
        bundle,
        base_policies={"single": base},
        boundary_config=_boundary_config("stack_cups"),
    )
    episode = load_low_dim_obs_pickles([path])[0]
    converted = make_unimanual_demonstrations(
        [episode],
        experiment_task("stack_cups").spec,
        names=[path.parent.name],
    ).demonstrations[0]
    dynamac = DynaMACObservation(
        converted.ee_pose[-1],
        {name: values[-1] for name, values in converted.frames.items()},
    )
    runtime = RuntimeObservation.from_dynamac(
        dynamac,
        tick=0,
        gripper_state=converted.gripper[-1],
    )
    policy.reset({"single": dynamac})
    model = policy.task_models["single"]
    final_state = max(model.states)
    mode = policy._mode_by_arm_skill["single"][final_state.skill_index]
    node = model.state(final_state)
    policy.belief_updaters["single"].reset(
        initial_progress={final_state: 1.0},
        initial_relations={
            frame: prior[mode].copy()
            for frame, prior in node.demo_relation_priors.items()
        },
    )
    policy.execution_controllers["single"].reset(final_state)

    assert policy.complete is False
    cycle = policy.act({"single": dynamac}, {"single": runtime})
    assert cycle.arms["single"].execution is not None
    assert cycle.arms["single"].execution.weighted_action.state_id == final_state
    assert policy.complete is False
    # This test isolates the terminal commit contract.  State-to-gripper
    # authorization itself is tested independently above and by real replay.
    authorized_arm = replace(
        cycle.arms["single"],
        command=replace(
            cycle.arms["single"].command,
            gripper_authorized=True,
        ),
    )
    policy._last_cycle = replace(cycle, arms={"single": authorized_arm})
    # The task-state model has already authorized the final discrete action;
    # executor ``progressed`` versus ``reached`` remains physical feedback and
    # cannot become a second policy-completion gate.
    policy.commit(
        task_command_applied=True,
        absolute_target_completed=False,
    )
    assert policy.complete is True
    assert policy._last_executed_reference == {"single": final_state}
    with pytest.raises(RuntimeError, match="已完成"):
        policy.act({"single": dynamac}, {"single": runtime})


def test_no_goal_recovery_servos_frozen_target_until_legal_reentry() -> None:
    bundle = BUNDLE_ROOT / "stack_cups"
    checkpoint = UNIMANUAL_BASE_ROOT / "stack_cups/model.npz"
    try:
        path = demonstration_paths(UNIMANUAL_DATA_ROOT, "stack_cups", 1)[0]
    except FileNotFoundError:
        pytest.skip("本地未安装当前 StackCups 正常示范")
    if not bundle.is_dir() or not checkpoint.is_file():
        pytest.skip("本地未安装当前 StackCups v5 bundle 或 DynaMAC checkpoint")

    base = DynaMAC.load(checkpoint)
    policy = TSFMultiStreamPolicy.load(
        bundle,
        base_policies={"single": base},
        boundary_config=_boundary_config("stack_cups"),
    )
    episode = load_low_dim_obs_pickles([path])[0]
    converted = make_unimanual_demonstrations(
        [episode],
        experiment_task("stack_cups").spec,
        names=[path.parent.name],
    ).demonstrations[0]
    initial = min(policy.task_models["single"].states)
    observed_pose = converted.ee_pose[0].copy()
    observed_pose[0] += 0.30
    dynamac = DynaMACObservation(
        observed_pose,
        {name: values[0] for name, values in converted.frames.items()},
    )
    runtime = RuntimeObservation.from_dynamac(
        dynamac,
        tick=0,
        gripper_state=converted.gripper[0],
    )
    policy.reset({"single": dynamac})
    policy.recovery_managers["single"].begin_recovery(
        RecoveryTriggerDecision(True, ("single:no_plausible_state",), ()),
        source_state=initial,
        mode=policy._mode_by_arm_skill["single"][initial.skill_index],
    )

    cycle = policy.act({"single": dynamac}, {"single": runtime})
    arm = cycle.arms["single"]
    assert arm.mode_before == ExecutionMode.RECOVERY
    assert arm.execution is None
    assert arm.command.source == "recovery_frozen_task_target"
    assert not np.allclose(arm.command.pose, observed_pose)
    # The command prepared for this cycle remains the frozen recovery target,
    # but complete-state recovery may legally re-enter a supported state for
    # the following TASK cycle.  That explicit re-entry is not normal progress
    # advancement and must agree with the committed cursor.
    assert arm.recovery is not None
    assert arm.recovery.reentry is not None
    assert arm.mode_after == ExecutionMode.TASK
    assert (
        policy.execution_controllers["single"].cursor.reference_state
        == arm.recovery.reentry.state_id
    )
    assert policy._last_executed_reference["single"] is None


def test_top_level_auxiliary_mode_freezes_peer_progress_and_boundaries() -> None:
    bundle = BUNDLE_ROOT / "bimanual_handover_item"
    checkpoints = {
        arm: BIMANUAL_BASE_ROOT / "bimanual_handover_item" / f"{arm}.npz"
        for arm in ("left", "right")
    }
    try:
        path = demonstration_paths(BIMANUAL_DATA_ROOT, "bimanual_handover_item", 1)[0]
    except FileNotFoundError:
        pytest.skip("本地未安装当前 HandOver 正常示范")
    if not bundle.is_dir() or not all(path.is_file() for path in checkpoints.values()):
        pytest.skip("本地未安装当前 HandOver v5 bundle 或 DynaMAC checkpoint")

    converted = make_bimanual_demonstrations(
        load_low_dim_obs_pickles([path]),
        "bimanual_handover_item",
        names=[path.parent.name],
    )
    demonstrations = {
        "left": converted.left_demonstrations[0],
        "right": converted.right_demonstrations[0],
    }
    observations = {}
    runtimes = {}
    for arm, peer in (("left", "right"), ("right", "left")):
        demonstration = demonstrations[arm]
        frames = {
            name: values[0].copy() for name, values in demonstration.frames.items()
        }
        frames[f"{peer}_ee"] = demonstrations[peer].ee_pose[0].copy()
        observation = DynaMACObservation(demonstration.ee_pose[0], frames)
        observations[arm] = observation
        runtimes[arm] = RuntimeObservation.from_dynamac(
            observation,
            tick=0,
            gripper_state=demonstration.gripper[0],
        )

    policy = TSFMultiStreamPolicy.load(
        bundle,
        base_policies={arm: DynaMAC.load(path) for arm, path in checkpoints.items()},
        boundary_config=_boundary_config("bimanual_handover_item"),
    )
    policy.reset(observations)
    initial = {
        arm: policy.execution_controllers[arm].cursor.reference_state
        for arm in policy.arms
    }
    right_mode = policy._mode_by_arm_skill["right"][initial["right"].skill_index]
    policy.recovery_managers["right"].begin_recovery(
        RecoveryTriggerDecision(True, ("right:no_plausible_state",), ()),
        source_state=initial["right"],
        mode=right_mode,
    )

    cycle = policy.act(observations, runtimes)

    left = cycle.arms["left"]
    assert left.mode_before == left.mode_after == ExecutionMode.TASK
    assert left.execution is None
    assert left.belief.update_sequence == ("frozen_progress", "relation_posterior")
    assert left.command.source.startswith("auxiliary_peer_frozen_task_target")
    assert (
        policy.execution_controllers["left"].cursor.reference_state == initial["left"]
    )
    # Recovery may evaluate its own legal reentry, but the TASK peer neither
    # contributes a normal request nor commits a normal boundary transaction.
    if cycle.boundary is not None:
        assert "left" not in cycle.boundary.requests
        assert cycle.boundary.transaction is None
    assert policy._last_executed_reference["left"] is None
