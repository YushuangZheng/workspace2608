"""Shared TSF policy/evaluator protocol identities."""

from __future__ import annotations

from typing import Any


TSF_GRIPPER_TIMING_PROTOCOL_ID = (
    "tsf-state-or-link-edge-gripper-v2"
)
TSF_GRIPPER_TIMING_RULE = (
    "state_command_or_learned_link_transition_preparation"
)


def tsf_gripper_timing_metadata() -> dict[str, Any]:
    return {
        "protocol_id": TSF_GRIPPER_TIMING_PROTOCOL_ID,
        "rule": TSF_GRIPPER_TIMING_RULE,
        "policy_pose_tick": "t",
        "default_gripper_tick": "t",
        "boundary_transition_gripper_tick": "commit_or_link_prepare",
        "preparation_scope": (
            "learned_link_or_link_pending_open_to_closed_edge_only"
        ),
        "preparation_progress_semantics": "no_stateid_or_localdone_commit",
        "release_semantics": "post_boundary_commit_only",
        "authorization": (
            "aligned_task_state_or_committed_boundary_or_link_edge_preparation"
        ),
        "executor_status_role": "physical_diagnostic_only",
        "task_specific_branches": False,
        "training_labels_modified": False,
    }


__all__ = [
    "TSF_GRIPPER_TIMING_PROTOCOL_ID",
    "TSF_GRIPPER_TIMING_RULE",
    "tsf_gripper_timing_metadata",
]
